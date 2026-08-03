"""把 W-07 train payload 适配为 typed logic candidate 与独立 Evidence。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    OBJECT_SET_EXPR,
    OBJECT_SPAN,
    OBJECT_VARIABLE,
    CorpusVersion,
    CurriculumVersion,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    occurrence_identity,
    span_identity,
)
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicOperatorCandidateProtocol,
    LogicOperatorCandidateSpec,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    ConjunctionOperator,
    ConditionOperator,
    DisjunctionOperator,
    ExistentialOperator,
    FiniteQuantifierDomain,
    LogicEvidenceState,
    LogicOperatorDefinition,
    ModalOperator,
    NegationOperator,
    OperatorSlot,
    QuantifierDefinition,
    UniversalOperator,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    describe_variable,
    semantic_source,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
    TypedValue,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_RESOURCE_BUDGET,
    W07_SUBSTAGE_ORDER,
)
from pure_integer_ai.experiments.ph2_w07_payload import W07TrainingPayload
from pure_integer_ai.experiments.ph2_w07_registry import (
    W07RegistryError,
    audit_w07_registry_payload,
)


W07_NAMESPACE = 70707
W07_IDENTITY_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(7),
)
W07_SCHEMA_REJECTION_REASONS = (
    "DOMAIN_TYPE_MISMATCH",
    "MISSING_INNER_OPERATOR",
)
_SOURCE_BEARING_KINDS = frozenset({
    OBJECT_BINDER,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE_BINDING,
    OBJECT_SET_EXPR,
    OBJECT_VARIABLE,
})
_SOURCE_PROTOCOL_NAMESPACES = {
    "NOT": 20701,
    "AND_OR": 20701,
    "CONDITION": 20701,
    "EXISTS": 20801,
    "FORALL": 20801,
    "MODAL": 20701,
    "NESTED_SCOPE": 20901,
}
_SOURCE_PROTOCOL_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)


class W07TypedAdapterError(ValueError):
    """W-07 typed payload、schema、source 或 Evidence 无法安全适配。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _identity(value: Any, *, where: str) -> ObjectIdentity:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W07TypedAdapterError(f"{where} 不是严格整数 identity key")
    try:
        return ObjectIdentity.from_stable_key(tuple(value))
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError(f"{where} identity 非法") from error


def _source(value: Any, *, where: str) -> SourceRef:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W07TypedAdapterError(f"{where} 不是严格整数 source key")
    try:
        return SourceRef.from_stable_key(tuple(value))
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError(f"{where} source 非法") from error


def _scope(value: Any, *, where: str) -> ScopeIdentity:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W07TypedAdapterError(f"{where} 不是严格整数 scope key")
    try:
        return ScopeIdentity.from_stable_key(tuple(value))
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError(f"{where} scope 非法") from error


def _int_key(value: Any, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W07TypedAdapterError(f"{where} 不是非空严格整数 key")
    return tuple(value)


def w07_logic_candidate_protocol() -> LogicOperatorCandidateProtocol:
    """返回七类 operator 共用的单一 H-05 图 predicate 协议。"""
    return LogicOperatorCandidateProtocol(
        concept_identity(
            (W07_NAMESPACE, 501), versions=W07_IDENTITY_VERSIONS),
        concept_identity(
            (W07_NAMESPACE, 502), versions=W07_IDENTITY_VERSIONS),
        concept_identity(
            (W07_NAMESPACE, 503), versions=W07_IDENTITY_VERSIONS),
    )


def _source_candidate_protocol(substage: str) -> LogicOperatorCandidateProtocol:
    """返回三个公开 compiler family 各自冻结的候选图协议。"""
    namespace = _SOURCE_PROTOCOL_NAMESPACES.get(substage)
    if namespace is None:
        raise W07TypedAdapterError("W-07 source protocol substage 未注册")
    return LogicOperatorCandidateProtocol(
        concept_identity((namespace, 5, 1), versions=_SOURCE_PROTOCOL_VERSIONS),
        concept_identity((namespace, 5, 2), versions=_SOURCE_PROTOCOL_VERSIONS),
        concept_identity((namespace, 5, 3), versions=_SOURCE_PROTOCOL_VERSIONS),
    )


def _anchor_source(anchor: ObjectIdentity) -> SourceRef:
    """从 Occurrence/Span 前缀恢复完整 SourceRef，拒绝类型和版本漂移。"""
    if anchor.object_kind not in {OBJECT_OCCURRENCE, OBJECT_SPAN}:
        raise W07TypedAdapterError("W-07 bound source anchor 类型非法")
    try:
        source = SourceRef.from_stable_key(tuple(anchor.components[:11]))
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError("W-07 bound source anchor 来源非法") from error
    try:
        if anchor.object_kind == OBJECT_OCCURRENCE:
            if len(anchor.components) != 14:
                raise ValueError("occurrence 长度非法")
            rebuilt = occurrence_identity(
                source,
                start=anchor.components[-3],
                end=anchor.components[-2],
                ordinal=anchor.components[-1],
            )
        else:
            if len(anchor.components) < 15:
                raise ValueError("span 长度非法")
            ordinal = anchor.components[11]
            size = anchor.components[12]
            values = anchor.components[13:]
            if type(size) is not int or size <= 0 or len(values) != size * 2:
                raise ValueError("span members 长度非法")
            rebuilt = span_identity(
                source,
                members=tuple(
                    (values[index], values[index + 1])
                    for index in range(0, len(values), 2)
                ),
                ordinal=ordinal,
            )
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError("W-07 bound source anchor 结构非法") from error
    if rebuilt != anchor:
        raise W07TypedAdapterError("W-07 bound source anchor owner/version 漂移")
    return source


def _validate_bound_source(root: BoundProposition, source: SourceRef) -> None:
    """递归核验 bound tree 中所有来源化身份都属于 forming source。"""
    def require(item: ObjectIdentity, *, where: str) -> None:
        try:
            actual = semantic_source(item)
        except (TypeError, ValueError) as error:
            raise W07TypedAdapterError(f"W-07 {where} 来源身份非法") from error
        if actual != source:
            raise W07TypedAdapterError(f"W-07 {where} 跨 forming source")

    def visit(item: BoundProposition) -> None:
        require(item.template, where="bound Proposition")
        require(item.context, where="bound context")
        if _anchor_source(item.source_anchor) != source:
            raise W07TypedAdapterError("W-07 bound source anchor 跨 forming source")
        for binder in item.introduced_binders:
            require(binder, where="bound Binder")
        for variable in item.applied_variables:
            require(variable, where="bound Variable")
        for binding in item.bindings:
            if isinstance(binding.filler, BoundProposition):
                visit(binding.filler)
            elif binding.filler.object_kind in _SOURCE_BEARING_KINDS:
                require(binding.filler, where="bound filler")

    visit(root)


def _decode_bound(value: Any, *, where: str = "bound_root") -> BoundProposition:
    """递归恢复不可物化 BoundProposition，保留精确 Role 与 Binder。"""
    fields = {
        "applied_variable_keys",
        "bindings",
        "context_key",
        "instruction_key",
        "introduced_binder_keys",
        "predicate_key",
        "source_anchor_key",
        "structure_key",
        "template_key",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise W07TypedAdapterError(f"{where} 字段集合漂移")
    raw_binders = value["introduced_binder_keys"]
    raw_variables = value["applied_variable_keys"]
    raw_bindings = value["bindings"]
    if (not isinstance(raw_binders, list)
            or not isinstance(raw_variables, list)
            or not isinstance(raw_bindings, list)):
        raise W07TypedAdapterError(f"{where} Binder/Variable/bindings 非列表")
    bindings = []
    for index, item in enumerate(raw_bindings):
        if not isinstance(item, dict) or set(item) != {
                "filler", "ordinal", "role_key"}:
            raise W07TypedAdapterError(f"{where}.bindings[{index}] 字段漂移")
        ordinal = item["ordinal"]
        if type(ordinal) is not int or ordinal < 0:
            raise W07TypedAdapterError(f"{where}.bindings[{index}] ordinal 非法")
        filler = item["filler"]
        if not isinstance(filler, dict) or filler.get("kind") not in {
                "identity", "bound_proposition"}:
            raise W07TypedAdapterError(f"{where}.bindings[{index}] filler 非法")
        if filler["kind"] == "identity":
            if set(filler) != {"identity_key", "kind"}:
                raise W07TypedAdapterError("identity filler 字段漂移")
            decoded_filler: ObjectIdentity | BoundProposition = _identity(
                filler["identity_key"], where="identity filler")
        else:
            if set(filler) != {"bound", "kind"}:
                raise W07TypedAdapterError("bound filler 字段漂移")
            decoded_filler = _decode_bound(
                filler["bound"], where=f"{where}.bindings[{index}].bound")
        bindings.append(BoundRoleBinding(
            _identity(item["role_key"], where="bound role"),
            decoded_filler,
            ordinal,
        ))
    try:
        return BoundProposition(
            _identity(value["template_key"], where=f"{where}.template"),
            _identity(value["instruction_key"], where=f"{where}.instruction"),
            _identity(value["predicate_key"], where=f"{where}.predicate"),
            _identity(value["structure_key"], where=f"{where}.structure"),
            _identity(value["source_anchor_key"], where=f"{where}.source_anchor"),
            _identity(value["context_key"], where=f"{where}.context"),
            tuple(_identity(item, where=f"{where}.binder")
                  for item in raw_binders),
            tuple(bindings),
            tuple(_identity(item, where=f"{where}.variable")
                  for item in raw_variables),
        )
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError(f"{where} typed schema 非法") from error


def _bound_by_template(root: BoundProposition) -> dict[ObjectIdentity, BoundProposition]:
    result: dict[ObjectIdentity, BoundProposition] = {}

    def visit(item: BoundProposition) -> None:
        prior = result.get(item.template)
        if prior is not None and prior != item:
            raise W07TypedAdapterError("同一 Proposition template 绑定不同 bound view")
        result[item.template] = item
        for binding in item.bindings:
            if isinstance(binding.filler, BoundProposition):
                visit(binding.filler)

    visit(root)
    return result


def _handler(operator_kind: int):
    handlers = {
        1: NegationOperator,
        2: ConjunctionOperator,
        3: DisjunctionOperator,
        4: ConditionOperator,
        5: ExistentialOperator,
        6: UniversalOperator,
        7: ModalOperator,
    }
    factory = handlers.get(operator_kind)
    if factory is None:
        raise W07TypedAdapterError("W-07 operator handler 未注册")
    return factory()


def _decode_protocol(value: Any) -> LogicOperatorCandidateProtocol:
    if not isinstance(value, dict) or set(value) != {
            "instruction_predicate_key", "slot_predicate_key",
            "structure_predicate_key"}:
        raise W07TypedAdapterError("W-07 candidate protocol 字段漂移")
    try:
        return LogicOperatorCandidateProtocol(
            _identity(value["structure_predicate_key"], where="structure predicate"),
            _identity(value["instruction_predicate_key"], where="instruction predicate"),
            _identity(value["slot_predicate_key"], where="slot predicate"),
        )
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError("W-07 source candidate protocol 非法") from error


def _decode_definition(value: Any, *, operator_kind: int) -> LogicOperatorDefinition:
    if not isinstance(value, dict) or set(value) != {
            "instruction_key", "slots", "structure_key"}:
        raise W07TypedAdapterError("W-07 operator definition 字段漂移")
    raw_slots = value["slots"]
    if not isinstance(raw_slots, list) or not raw_slots:
        raise W07TypedAdapterError("W-07 operator slots 为空")
    slots = []
    for item in raw_slots:
        if not isinstance(item, dict) or set(item) != {"ordinal", "role_key"}:
            raise W07TypedAdapterError("W-07 operator slot 字段漂移")
        if type(item["ordinal"]) is not int or item["ordinal"] < 0:
            raise W07TypedAdapterError("W-07 operator slot ordinal 非法")
        slots.append(OperatorSlot(
            _identity(item["role_key"], where="operator role"),
            item["ordinal"],
        ))
    try:
        return LogicOperatorDefinition(
            _identity(value["structure_key"], where="operator structure"),
            _identity(value["instruction_key"], where="operator instruction"),
            tuple(slots),
            _handler(operator_kind),
        )
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError("W-07 operator definition 非法") from error


def _decode_spec(
        value: Any,
        definition: LogicOperatorDefinition,
        ) -> LogicOperatorCandidateSpec:
    if not isinstance(value, dict) or set(value) != {
            "candidate_key", "competition_key", "forming_source_keys",
            "graph_binding_count"}:
        raise W07TypedAdapterError("W-07 candidate spec 字段漂移")
    sources = value["forming_source_keys"]
    if not isinstance(sources, list) or not sources:
        raise W07TypedAdapterError("W-07 forming source 为空")
    try:
        spec = LogicOperatorCandidateSpec(
            _identity(value["candidate_key"], where="candidate Proposition"),
            definition,
            _int_key(value["competition_key"], where="competition key"),
            tuple(_source(item, where="forming source") for item in sources),
        )
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError("W-07 candidate spec 非法") from error
    protocol = w07_logic_candidate_protocol()
    if value["graph_binding_count"] != len(
            spec.candidate_definition(protocol).bindings):
        raise W07TypedAdapterError("W-07 candidate graph binding count 漂移")
    return spec


def _validate_spec_bound(
        spec: LogicOperatorCandidateSpec,
        bound: BoundProposition,
        ) -> None:
    if (spec.candidate != bound.template
            or spec.definition.structure != bound.structure
            or spec.definition.instruction != bound.instruction):
        raise W07TypedAdapterError("W-07 candidate 与 bound Proposition 身份错配")
    definition_slots = tuple(
        (item.role, item.ordinal) for item in spec.definition.slots)
    bound_slots = tuple(
        (item.role, item.ordinal) for item in bound.bindings)
    if definition_slots != bound_slots:
        raise W07TypedAdapterError("W-07 candidate ordered Role slots 漂移")


@dataclass(frozen=True)
class W07QuantifierValueEvidence:
    """一个 domain value 的独立四态内容 Evidence。"""

    value: ObjectIdentity
    state: LogicEvidenceState


@dataclass(frozen=True)
class W07QuantifierBinding:
    """一个已验证 Binder/Variable/domain/body 的 quantifier resolver 输入。"""

    operator_candidate: ObjectIdentity
    definition: QuantifierDefinition
    value_role: ObjectIdentity
    value_evidence: tuple[W07QuantifierValueEvidence, ...]


def _variable_roles(
        bound: BoundProposition,
        variable: ObjectIdentity,
        ) -> tuple[ObjectIdentity, ...]:
    """返回 Variable 在量词 body 子树中的精确 Role 路径端点。"""
    roles = []

    def visit(item: BoundProposition) -> None:
        for binding in item.bindings:
            if binding.filler == variable:
                roles.append(binding.role)
            elif isinstance(binding.filler, BoundProposition):
                visit(binding.filler)

    visit(bound)
    return tuple(roles)


def _decode_quantifier(
        value: Any,
        spec: LogicOperatorCandidateSpec,
        bound: BoundProposition,
        *,
        raw_value_evidence: Any,
        raw_value_role: Any = None,
        ) -> W07QuantifierBinding:
    fields = {
        "binder_key", "body_slot", "domain", "value_type_key", "variable_key"}
    if not isinstance(value, dict) or set(value) != fields:
        raise W07TypedAdapterError("W-07 quantifier definition 字段漂移")
    body_slot = value["body_slot"]
    domain_value = value["domain"]
    if (not isinstance(body_slot, dict)
            or set(body_slot) != {"ordinal", "role_key"}
            or not isinstance(domain_value, dict)
            or set(domain_value) != {
                "closed", "closure_evidence_keys", "domain_key", "values"}):
        raise W07TypedAdapterError("W-07 quantifier body/domain 字段漂移")
    value_type = _identity(value["value_type_key"], where="quantifier value type")
    binder = _identity(value["binder_key"], where="quantifier Binder")
    variable = _identity(value["variable_key"], where="quantifier Variable")
    descriptor = describe_variable(variable)
    if descriptor.binder != binder or descriptor.value_type != value_type:
        raise W07TypedAdapterError("W-07 quantifier Variable 声明漂移")
    raw_values = domain_value["values"]
    if not isinstance(raw_values, list):
        raise W07TypedAdapterError("W-07 quantifier domain values 非列表")
    typed_values = []
    domain_type_mismatch = False
    for item in raw_values:
        if not isinstance(item, dict) or set(item) != {"type_key", "value_key"}:
            raise W07TypedAdapterError("W-07 quantifier domain value 字段漂移")
        actual_type = _identity(item["type_key"], where="domain actual type")
        if actual_type != value_type:
            domain_type_mismatch = True
        typed_values.append(TypedValue(
            _identity(item["value_key"], where="domain value"), actual_type))
    closed = domain_value["closed"]
    if closed not in {0, 1} or type(closed) is not int:
        raise W07TypedAdapterError("W-07 quantifier domain closed 非 bit")
    try:
        domain = FiniteQuantifierDomain(
            _identity(domain_value["domain_key"], where="quantifier domain"),
            tuple(typed_values),
            bool(closed),
            tuple(_identity(item, where="domain closure Evidence")
                  for item in domain_value["closure_evidence_keys"]),
        )
        definition = QuantifierDefinition(
            binder,
            variable,
            OperatorSlot(
                _identity(body_slot["role_key"], where="quantifier body role"),
                body_slot["ordinal"],
            ),
            domain,
        )
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError("W-07 quantifier typed definition 非法") from error
    if (binder not in bound.introduced_binders
            or definition.body_slot != spec.definition.slots[0]):
        raise W07TypedAdapterError("W-07 quantifier Binder/body scope 漂移")
    bound_source = semantic_source(bound.template)
    source_items = (
        binder,
        variable,
        domain.domain,
        *(item.value for item in domain.values),
        *domain.closure_evidence,
    )
    try:
        source_drift = any(
            semantic_source(item) != bound_source for item in source_items)
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError(
            "W-07 quantifier Binder/domain/value 来源非法") from error
    if source_drift:
        raise W07TypedAdapterError("W-07 quantifier Binder/domain/value 来源漂移")
    variable_roles = _variable_roles(bound, variable)
    if len(variable_roles) != 1:
        raise W07TypedAdapterError("W-07 quantifier Variable 未唯一绑定到 body Role")
    value_role = variable_roles[0]
    if (raw_value_role is not None
            and _identity(raw_value_role, where="quantifier value role")
            != value_role):
        raise W07TypedAdapterError("W-07 quantifier value Role 漂移")
    if (not isinstance(raw_value_evidence, list)
            or len(raw_value_evidence) != len(domain.values)):
        raise W07TypedAdapterError("W-07 quantifier value Evidence 数量漂移")
    decoded_evidence = []
    for index, (raw_evidence, typed_value) in enumerate(zip(
            raw_value_evidence, domain.values)):
        if (not isinstance(raw_evidence, dict)
                or set(raw_evidence) != {"refute", "support", "value_key"}):
            raise W07TypedAdapterError(
                f"W-07 quantifier value Evidence[{index}] 字段漂移")
        support = raw_evidence["support"]
        refute = raw_evidence["refute"]
        if (type(support) is not int or support not in {0, 1}
                or type(refute) is not int or refute not in {0, 1}
                or _identity(
                    raw_evidence["value_key"],
                    where="quantifier Evidence value",
                ) != typed_value.value):
            raise W07TypedAdapterError(
                f"W-07 quantifier value Evidence[{index}] 四态/顺序漂移")
        decoded_evidence.append(W07QuantifierValueEvidence(
            typed_value.value,
            LogicEvidenceState(bool(support), bool(refute)),
        ))
    if domain_type_mismatch:
        raise W07TypedAdapterError("DOMAIN_TYPE_MISMATCH")
    return W07QuantifierBinding(
        spec.candidate,
        definition,
        value_role,
        tuple(decoded_evidence),
    )


@dataclass(frozen=True)
class W07ModalResolutionPlan:
    """保留 modal resolver 的来源、输入/输出 scope、四态和证据。"""

    operator_candidate: ObjectIdentity
    status: str
    state: LogicEvidenceState
    source: SourceRef
    input_scope: ScopeIdentity
    output_scope: ScopeIdentity | None
    evidence_ids: tuple[int, ...]


def _decode_modal_plan(
        value: Any,
        spec: LogicOperatorCandidateSpec,
        request_scope: ScopeIdentity,
        ) -> W07ModalResolutionPlan:
    fields = {
        "evidence_ids", "input_scope_key", "output_scope_key",
        "resolution_state", "source_key", "source_unchanged", "status"}
    if not isinstance(value, dict) or set(value) != fields:
        raise W07TypedAdapterError("W-07 modal resolution plan 字段漂移")
    state = value["resolution_state"]
    if (not isinstance(state, dict) or set(state) != {"refute", "support"}
            or state["support"] not in {0, 1}
            or state["refute"] not in {0, 1}
            or type(state["support"]) is not int
            or type(state["refute"]) is not int):
        raise W07TypedAdapterError("W-07 modal resolution 四态 bit 非法")
    evidence_ids = value["evidence_ids"]
    if (not isinstance(evidence_ids, list)
            or any(type(item) is not int or item <= 0 for item in evidence_ids)):
        raise W07TypedAdapterError("W-07 modal evidence ids 非法")
    source = _source(value["source_key"], where="modal source")
    input_scope = _scope(value["input_scope_key"], where="modal input scope")
    raw_output_scope = value["output_scope_key"]
    output_scope = (
        None if raw_output_scope is None
        else _scope(raw_output_scope, where="modal output scope")
    )
    if (value["source_unchanged"] != 1
            or input_scope != request_scope
            or input_scope.source != source
            or (output_scope is not None and output_scope.source != source)):
        raise W07TypedAdapterError("W-07 modal source/scope 漂移")
    status = value["status"]
    if status not in {"RESOLVED", "MISSING", "DENIED", "BUDGET_UNDECIDED"}:
        raise W07TypedAdapterError("W-07 modal resolver status 未注册")
    if status == "RESOLVED" and (output_scope is None or not evidence_ids):
        raise W07TypedAdapterError("W-07 resolved modal 缺输出 scope/Evidence")
    if status != "RESOLVED" and (
            state != {"refute": 0, "support": 0}
            or output_scope is not None
            or evidence_ids):
        raise W07TypedAdapterError("W-07 未决 modal resolver 不得携带现实四态")
    return W07ModalResolutionPlan(
        spec.candidate,
        status,
        LogicEvidenceState(bool(state["support"]), bool(state["refute"])),
        source,
        input_scope,
        output_scope,
        tuple(evidence_ids),
    )


@dataclass(frozen=True)
class W07SourceBinding:
    """把公开 SourceRefRecord 绑定到 compiler 形成的 typed SourceRef。"""

    record: SourceRefRecord
    source_ref: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.record, SourceRefRecord):
            raise TypeError("W-07 source binding record 类型非法")
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("W-07 source binding source_ref 类型非法")
        if (self.source_ref.versions.parser.value != self.record.parser_version
                or self.source_ref.versions.curriculum.value
                != self.record.course_version):
            raise W07TypedAdapterError(
                "W-07 SourceRefRecord 与 typed SourceRef version 漂移")


@dataclass(frozen=True)
class W07LogicProposal:
    """一个通过全量 schema 校验的来源化 logic observation。"""

    observation: ObservationRecord
    source_binding: W07SourceBinding
    source_protocol: LogicOperatorCandidateProtocol
    specs: tuple[LogicOperatorCandidateSpec, ...]
    bound_root: BoundProposition
    request_scope: ScopeIdentity
    operator_families: tuple[str, ...]
    quantifiers: tuple[W07QuantifierBinding, ...]
    modal_plans: tuple[W07ModalResolutionPlan, ...]

    def __post_init__(self) -> None:
        if not self.specs or not self.operator_families:
            raise W07TypedAdapterError("W-07 proposal specs/operator family 为空")
        if len(self.specs) != len(self.operator_families):
            raise W07TypedAdapterError("W-07 proposal spec/family 数量错配")
        if any(item.forming_sources != (self.source_binding.source_ref,)
               for item in self.specs):
            raise W07TypedAdapterError("W-07 proposal forming source 漂移")
        if self.request_scope.source != self.source_binding.source_ref:
            raise W07TypedAdapterError("W-07 proposal request scope/source 漂移")
        if (self.source_binding.record.stable_key
                != self.observation.source_ref_key
                or self.source_binding.record.license_id
                != self.observation.license_partition):
            raise W07TypedAdapterError("W-07 proposal SourceRefRecord 引用漂移")
        if self.source_protocol != _source_candidate_protocol(
                self.observation.substage):
            raise W07TypedAdapterError("W-07 proposal source protocol 漂移")


@dataclass(frozen=True)
class W07SchemaRejection:
    """保留无效 observation 的来源、teacher 与精确拒绝原因。"""

    observation: ObservationRecord
    source_record: SourceRefRecord
    teacher_record: TeacherEvidenceRecord
    reason: str

    def __post_init__(self) -> None:
        if self.reason not in W07_SCHEMA_REJECTION_REASONS:
            raise W07TypedAdapterError("W-07 schema rejection reason 未注册")
        if (self.observation.perturbation_kind != self.reason
                or self.source_record.stable_key
                != self.observation.source_ref_key
                or self.teacher_record.observation_key
                != self.observation.stable_key
                or self.teacher_record.source_ref_key
                != self.observation.source_ref_key):
            raise W07TypedAdapterError("W-07 schema rejection 来源/原因引用漂移")


@dataclass(frozen=True)
class W07EvidenceBinding:
    """把结构采用立场与表达式内容四态分账绑定到合法 proposal。"""

    teacher_record: TeacherEvidenceRecord
    proposal: W07LogicProposal
    stances: tuple[int, ...]
    content_stances: tuple[int, ...]
    expected_state: str
    expected_payload: CanonicalJsonObject
    reason_key: tuple[int, ...]
    supersedes_observation_key: StableRecordKey | None

    def __post_init__(self) -> None:
        expected = {
            "TRUE": (EVIDENCE_SUPPORT,),
            "FALSE": (EVIDENCE_REFUTE,),
            "UNKNOWN": (EVIDENCE_UNKNOWN,),
            "CONFLICT": (EVIDENCE_SUPPORT, EVIDENCE_REFUTE),
        }
        if (self.teacher_record.observation_key
                != self.proposal.observation.stable_key
                or self.teacher_record.source_ref_key
                != self.proposal.observation.source_ref_key
                or self.content_stances != expected.get(self.expected_state)
                or self.supersedes_observation_key
                != self.proposal.observation.supersedes_key):
            raise W07TypedAdapterError("W-07 Evidence binding 引用或内容四态漂移")
        if (not self.stances
                or any(item not in {
                    EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}
                    for item in self.stances)
                or not isinstance(self.expected_payload, CanonicalJsonObject)
                or not isinstance(self.reason_key, tuple)
                or not self.reason_key
                or any(type(item) is not int for item in self.reason_key)):
            raise W07TypedAdapterError("W-07 Evidence binding typed 字段非法")


@dataclass(frozen=True)
class W07TypedAdapterOutput:
    """W-07 纯 adapter 的 proposals/Evidence/rejections 分账。"""

    protocol: LogicOperatorCandidateProtocol
    source_bindings: tuple[W07SourceBinding, ...]
    proposals: tuple[W07LogicProposal, ...]
    specs: tuple[LogicOperatorCandidateSpec, ...]
    evidence: tuple[W07EvidenceBinding, ...]
    rejections: tuple[W07SchemaRejection, ...]
    record_count: int
    logic_operations: int

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            "evidence": [{
                "content_stances": list(item.content_stances),
                "expected_payload": item.expected_payload.to_value(),
                "expected_state": item.expected_state,
                "reason_key": list(item.reason_key),
                "stances": list(item.stances),
                "supersedes": (
                    None if item.supersedes_observation_key is None
                    else list(item.supersedes_observation_key.stable_key())
                ),
                "teacher": item.teacher_record.to_dict(),
            } for item in self.evidence],
            "logic_operations": self.logic_operations,
            "proposals": [{
                "bound_root": list(item.bound_root.stable_key()),
                "families": list(item.operator_families),
                "modal_plans": [{
                    "candidate": list(plan.operator_candidate.stable_key()),
                    "evidence_ids": list(plan.evidence_ids),
                    "input_scope": list(plan.input_scope.stable_key()),
                    "output_scope": (
                        None if plan.output_scope is None
                        else list(plan.output_scope.stable_key())
                    ),
                    "source": list(plan.source.stable_key()),
                    "state": list(plan.state.stable_key()),
                    "status": plan.status,
                } for plan in item.modal_plans],
                "observation": item.observation.to_dict(),
                "quantifiers": [{
                    "binder": list(value.definition.binder.stable_key()),
                    "body_slot": list(value.definition.body_slot.stable_key()),
                    "candidate": list(value.operator_candidate.stable_key()),
                    "domain": list(value.definition.domain.stable_key()),
                    "value_evidence": [{
                        "state": list(evidence.state.stable_key()),
                        "value": list(evidence.value.stable_key()),
                    } for evidence in value.value_evidence],
                    "value_role": list(value.value_role.stable_key()),
                    "variable": list(value.definition.variable.stable_key()),
                } for value in item.quantifiers],
                "request_scope": list(item.request_scope.stable_key()),
                "source_protocol": list(item.source_protocol.stable_key()),
                "specs": [
                    list(value.stable_key(self.protocol))
                    for value in item.specs
                ],
            } for item in self.proposals],
            "protocol": list(self.protocol.stable_key()),
            "record_count": self.record_count,
            "rejections": [{
                "observation": item.observation.to_dict(),
                "reason": item.reason,
                "source": item.source_record.to_dict(),
                "teacher": item.teacher_record.to_dict(),
            } for item in self.rejections],
            "source_bindings": [{
                "record": item.record.to_dict(),
                "source_ref": list(item.source_ref.stable_key()),
            } for item in self.source_bindings],
            "specs": [list(item.stable_key(self.protocol)) for item in self.specs],
        })


def _teacher_binding(
        proposal: W07LogicProposal,
        teacher: TeacherEvidenceRecord,
        ) -> W07EvidenceBinding:
    value = teacher.typed_evidence.to_value()
    if (teacher.observation_key != proposal.observation.stable_key
            or teacher.source_ref_key != proposal.observation.source_ref_key
            or teacher.visible_from_stage != "W-07"
            or not isinstance(value, dict)
            or set(value) != {"expected_payload", "expected_state", "seed_id"}):
        raise W07TypedAdapterError("W-07 TeacherEvidence 引用或字段漂移")
    mapping = {
        "TRUE": (EVIDENCE_SUPPORT,),
        "FALSE": (EVIDENCE_REFUTE,),
        "UNKNOWN": (EVIDENCE_UNKNOWN,),
        "CONFLICT": (EVIDENCE_SUPPORT, EVIDENCE_REFUTE),
    }
    content_stances = mapping.get(value["expected_state"])
    expected_payload = value["expected_payload"]
    if content_stances is None or not isinstance(expected_payload, dict):
        raise W07TypedAdapterError("W-07 TeacherEvidence 四态/payload 非法")
    decision = expected_payload.get("decision")
    if not isinstance(decision, str) or not decision:
        raise W07TypedAdapterError("W-07 TeacherEvidence 缺 typed decision")
    if decision.startswith("reject_"):
        stances = (EVIDENCE_REFUTE,)
    elif value["expected_state"] == "UNKNOWN":
        stances = (EVIDENCE_UNKNOWN,)
    elif value["expected_state"] == "CONFLICT":
        stances = (EVIDENCE_SUPPORT, EVIDENCE_REFUTE)
    else:
        stances = (EVIDENCE_SUPPORT,)
    reason = (
        W07_NAMESPACE,
        1,
        *_pack(teacher.stable_key.stable_key()),
        *_pack(proposal.observation.stable_key.stable_key()),
        proposal.observation.logical_order,
    )
    return W07EvidenceBinding(
        teacher,
        proposal,
        stances,
        content_stances,
        value["expected_state"],
        CanonicalJsonObject.from_value(expected_payload),
        reason,
        proposal.observation.supersedes_key,
    )


def _adapt_proposal(
        observation: ObservationRecord,
        source_record: SourceRefRecord,
        ) -> W07LogicProposal:
    value = observation.typed_payload.to_value()
    source_protocol = _decode_protocol(value.get("candidate_protocol"))
    if source_protocol != _source_candidate_protocol(observation.substage):
        raise W07TypedAdapterError("W-07 source candidate protocol 与 substage 漂移")
    bound_root = _decode_bound(value.get("bound_root"))
    bound_map = _bound_by_template(bound_root)
    request = value.get("consumer_request")
    if not isinstance(request, dict) or not isinstance(request.get("budget"), dict):
        raise W07TypedAdapterError("W-07 consumer request 非法")
    request_scope = _scope(request.get("scope_key"), where="request scope")
    root_key = _identity(request.get("root_key"), where="request root")
    if root_key != bound_root.template:
        raise W07TypedAdapterError("W-07 request root 与 bound root 漂移")
    try:
        source_ref = semantic_source(bound_root.template)
    except (TypeError, ValueError) as error:
        raise W07TypedAdapterError("W-07 bound root source 非法") from error
    if (request_scope.source != source_ref
            or source_ref.source_kind
            != _SOURCE_PROTOCOL_NAMESPACES[observation.substage] // 100):
        raise W07TypedAdapterError("W-07 request/source compiler family 漂移")
    _validate_bound_source(bound_root, source_ref)
    if source_record.license_id != observation.license_partition:
        raise W07TypedAdapterError("W-07 SourceRefRecord 与 Observation 许可漂移")
    source_binding = W07SourceBinding(source_record, source_ref)

    specs = []
    families = []
    quantifiers = []
    modal_plans = []
    if observation.substage != "NESTED_SCOPE":
        operator_kind = value.get("operator_kind")
        family = value.get("operator_family")
        if type(operator_kind) is not int or not isinstance(family, str):
            raise W07TypedAdapterError("W-07 operator kind/family 缺失")
        definition = _decode_definition(
            value.get("operator_definition"), operator_kind=operator_kind)
        spec = _decode_spec(value.get("candidate_spec"), definition)
        _validate_spec_bound(spec, bound_root)
        if spec.forming_sources != (source_ref,):
            raise W07TypedAdapterError("W-07 candidate/forming source 漂移")
        specs.append(spec)
        families.append(family)
        if observation.substage in {"EXISTS", "FORALL"}:
            quantifiers.append(_decode_quantifier(
                value.get("quantifier_definition"),
                spec,
                bound_root,
                raw_value_evidence=value.get("value_evidence"),
            ))
        if observation.substage == "MODAL":
            modal_plans.append(_decode_modal_plan(
                value.get("modal_resolution_plan"), spec, request_scope))
    else:
        layers = value.get("layers")
        derivation_order = value.get("derivation_order")
        if (not isinstance(layers, list) or len(layers) < 2
                or not isinstance(derivation_order, list)
                or len(derivation_order) != len(layers)):
            raise W07TypedAdapterError("W-07 nested layer/derivation 不完整")
        layer_ids = []
        missing_candidate = False
        for layer in layers:
            if not isinstance(layer, dict):
                raise W07TypedAdapterError("W-07 nested layer 类型错误")
            candidate_available = layer.get("candidate_available")
            if candidate_available not in {0, 1} or type(candidate_available) is not int:
                raise W07TypedAdapterError(
                    "W-07 nested candidate availability 非 bit")
            missing_candidate = missing_candidate or candidate_available == 0
            operator_kind = layer.get("operator_kind")
            family = layer.get("operator_family")
            layer_id = layer.get("layer_id")
            if (type(operator_kind) is not int or not isinstance(family, str)
                    or not isinstance(layer_id, str) or not layer_id):
                raise W07TypedAdapterError("W-07 nested layer identity 非法")
            definition = _decode_definition(
                layer.get("operator_definition"), operator_kind=operator_kind)
            spec = _decode_spec(layer.get("candidate_spec"), definition)
            bound_key = _identity(layer.get("bound_key"), where="nested bound key")
            if spec.candidate != bound_key or bound_key not in bound_map:
                raise W07TypedAdapterError("W-07 nested layer bound identity 漂移")
            _validate_spec_bound(spec, bound_map[bound_key])
            if spec.forming_sources != (source_ref,):
                raise W07TypedAdapterError("W-07 nested candidate/forming source 漂移")
            specs.append(spec)
            families.append(family)
            layer_ids.append(layer_id)
            quantifier_value = layer.get("quantifier_definition")
            if operator_kind in {5, 6}:
                base_fields = {
                    "binder_key", "body_slot", "domain",
                    "value_type_key", "variable_key",
                }
                extended_fields = base_fields | {
                    "value_evidence", "value_role_key"}
                if (not isinstance(quantifier_value, dict)
                        or set(quantifier_value) != extended_fields):
                    raise W07TypedAdapterError(
                        "W-07 nested quantifier definition 字段漂移")
                quantifiers.append(_decode_quantifier(
                    {key: quantifier_value[key] for key in base_fields},
                    spec,
                    bound_map[bound_key],
                    raw_value_evidence=quantifier_value["value_evidence"],
                    raw_value_role=quantifier_value["value_role_key"],
                ))
            elif quantifier_value is not None:
                raise W07TypedAdapterError("非 quantifier layer 混入 Binder/domain")
            modal_value = layer.get("modal_resolution_plan")
            if operator_kind == 7:
                modal_plans.append(_decode_modal_plan(
                    modal_value, spec, request_scope))
            elif modal_value is not None:
                raise W07TypedAdapterError("非 modal layer 混入 resolver plan")
        if derivation_order != list(reversed(layer_ids)):
            raise W07TypedAdapterError("W-07 nested derivation order 非内到外")
        if missing_candidate:
            raise W07TypedAdapterError("MISSING_INNER_OPERATOR")

    if any(spec.forming_sources != (source_ref,) for spec in specs):
        raise W07TypedAdapterError("W-07 observation 跨多个 forming source")
    return W07LogicProposal(
        observation,
        source_binding,
        source_protocol,
        tuple(specs),
        bound_root,
        request_scope,
        tuple(families),
        tuple(quantifiers),
        tuple(modal_plans),
    )


def _resource_budget(value: dict[str, int] | None) -> dict[str, int]:
    if value is None:
        return dict(W07_RESOURCE_BUDGET)
    if (not isinstance(value, dict) or set(value) != set(W07_RESOURCE_BUDGET)
            or any(type(item) is not int or item <= 0 for item in value.values())
            or any(value[key] > W07_RESOURCE_BUDGET[key] for key in value)):
        raise W07TypedAdapterError("W-07 adapter resource budget 非法或被放宽")
    return dict(value)


def adapt_w07_training_payload(
        payload: W07TrainingPayload,
        *,
        resource_budget: dict[str, int] | None = None,
        ) -> W07TypedAdapterOutput:
    """纯适配七 pack；无效 schema 分账且不形成 spec/Evidence。"""
    if not isinstance(payload, W07TrainingPayload):
        raise TypeError("W-07 adapter 只接受 W07TrainingPayload")
    budget = _resource_budget(resource_budget)
    record_count = (
        len(payload.source_refs) + len(payload.observations)
        + len(payload.teacher_evidence))
    if record_count > budget["max_records"]:
        raise W07TypedAdapterError("W-07 adapter record resource 超限")
    try:
        audit_w07_registry_payload(payload)
    except W07RegistryError as error:
        raise W07TypedAdapterError(
            f"W-07 registry payload 审计失败：{error}") from error
    sources = {item.stable_key: item for item in payload.source_refs}
    observations = {item.stable_key: item for item in payload.observations}
    if (len(sources) != len(payload.source_refs)
            or len(observations) != len(payload.observations)):
        raise W07TypedAdapterError("W-07 source/Observation stable key 重复")
    teachers: dict[StableRecordKey, TeacherEvidenceRecord] = {}
    teacher_routes = set()
    for teacher in payload.teacher_evidence:
        if teacher.observation_key in teachers:
            raise W07TypedAdapterError("W-07 Observation 有多个 TeacherEvidence")
        route = teacher.stable_key.stable_key()
        if route in teacher_routes:
            raise W07TypedAdapterError("W-07 TeacherEvidence stable key 重复")
        teacher_routes.add(route)
        teachers[teacher.observation_key] = teacher
    if set(teachers) != set(observations):
        raise W07TypedAdapterError("W-07 Observation/TeacherEvidence 不闭合")

    proposals = []
    evidence = []
    rejections = []
    source_bindings: dict[StableRecordKey, W07SourceBinding] = {}
    for observation in sorted(
            observations.values(),
            key=lambda item: (
                W07_SUBSTAGE_ORDER.index(item.substage),
                item.logical_order,
                item.stable_key.stable_key(),
            )):
        source_record = sources.get(observation.source_ref_key)
        teacher = teachers[observation.stable_key]
        if source_record is None:
            raise W07TypedAdapterError("W-07 Observation 缺 SourceRefRecord")
        try:
            proposal = _adapt_proposal(observation, source_record)
        except W07TypedAdapterError as error:
            reason = str(error)
            if (reason not in W07_SCHEMA_REJECTION_REASONS
                    or observation.perturbation_kind != reason):
                raise
            rejections.append(W07SchemaRejection(
                observation, source_record, teacher, reason))
            continue
        proposals.append(proposal)
        prior_source = source_bindings.get(observation.source_ref_key)
        if prior_source is not None and prior_source != proposal.source_binding:
            raise W07TypedAdapterError(
                "W-07 SourceRefRecord 绑定多个 typed SourceRef")
        source_bindings[observation.source_ref_key] = proposal.source_binding
        evidence.append(_teacher_binding(proposal, teacher))

    specs_by_candidate: dict[ObjectIdentity, LogicOperatorCandidateSpec] = {}
    for proposal in proposals:
        for spec in proposal.specs:
            prior = specs_by_candidate.get(spec.candidate)
            if prior is not None and prior != spec:
                raise W07TypedAdapterError("同一 W-07 candidate 绑定不同 spec")
            specs_by_candidate[spec.candidate] = spec
    rejected_candidates = {
        item.observation.stable_key for item in rejections}
    if any(item.proposal.observation.stable_key in rejected_candidates
           for item in evidence):
        raise W07TypedAdapterError("W-07 schema rejection 混入 Evidence")
    logic_operations = (
        len(specs_by_candidate) * 3 + len(evidence) + len(rejections))
    if logic_operations > budget["max_logic_operations"]:
        raise W07TypedAdapterError("W-07 adapter logic operation resource 超限")
    protocol = w07_logic_candidate_protocol()
    source_protocols = {
        item.source_protocol for item in proposals
    }
    expected_source_protocols = {
        _source_candidate_protocol(item) for item in W07_SUBSTAGE_ORDER
    }
    if source_protocols != expected_source_protocols:
        raise W07TypedAdapterError("W-07 三个 source candidate protocol 覆盖漂移")
    specs = tuple(sorted(
        specs_by_candidate.values(), key=lambda item: item.candidate.stable_key()))
    return W07TypedAdapterOutput(
        protocol,
        tuple(sorted(
            source_bindings.values(),
            key=lambda item: item.record.stable_key.stable_key())),
        tuple(proposals),
        specs,
        tuple(evidence),
        tuple(rejections),
        record_count,
        logic_operations,
    )


__all__ = [
    "W07EvidenceBinding",
    "W07LogicProposal",
    "W07ModalResolutionPlan",
    "W07QuantifierBinding",
    "W07QuantifierValueEvidence",
    "W07SchemaRejection",
    "W07SourceBinding",
    "W07TypedAdapterError",
    "W07TypedAdapterOutput",
    "W07_IDENTITY_VERSIONS",
    "W07_NAMESPACE",
    "adapt_w07_training_payload",
    "w07_logic_candidate_protocol",
]
