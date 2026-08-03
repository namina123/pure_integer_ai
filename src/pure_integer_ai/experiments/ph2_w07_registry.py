"""W-07 七个逻辑子阶段的来源、schema、证据与 consumer 注册表。"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w07_contract import W07_SUBSTAGE_ORDER
from pure_integer_ai.experiments.ph2_w07_payload import W07TrainingPayload


W07_CONSUMER_KEYS = ("UNDERSTANDING", "REASONING", "GENERATION")
W07_REQUIRED_SAMPLE_ROLES = ("support", "refute", "conflict", "supersede")
W07_EXPECTED_STATES = ("TRUE", "FALSE", "UNKNOWN", "CONFLICT")
W07_AUTHORED_SOURCE = "AUTHORED_CC0_V1"
W07_LICENSE_ID = "CC0-1.0"


class W07RegistryError(RuntimeError):
    """W-07 operator、来源、schema、证据或 consumer 未进入冻结注册表。"""


@dataclass(frozen=True)
class W07OperatorRegistryEntry:
    """冻结一个 W-07 子阶段的 typed operator 与 U/R/G 合同。"""

    substage_key: str
    operator_families: tuple[str, ...]
    operator_kinds: tuple[int, ...]
    payload_kind: str
    query_kind: str
    teacher_evidence_kind: str
    arities: tuple[int, ...]
    required_perturbations: tuple[str, ...]
    certificate_kinds: tuple[str, ...]
    consumer_keys: tuple[str, ...] = W07_CONSUMER_KEYS
    forming_source_keys: tuple[str, ...] = (W07_AUTHORED_SOURCE,)
    teacher_withdrawal_level: int = 0

    def __post_init__(self) -> None:
        if self.substage_key not in W07_SUBSTAGE_ORDER:
            raise W07RegistryError("W-07 registry substage 未冻结")
        sizes = tuple(map(len, (
            self.operator_families, self.operator_kinds, self.arities)))
        if not sizes[0] or len(set(self.operator_families)) != sizes[0]:
            raise W07RegistryError("W-07 operator family 重复或为空")
        if sizes[0] != sizes[1] or sizes[0] != sizes[2]:
            raise W07RegistryError("W-07 operator family/kind/arity 未对齐")
        if (any(type(item) is not int or item <= 0 for item in self.operator_kinds)
                or any(type(item) is not int or item <= 0 for item in self.arities)):
            raise W07RegistryError("W-07 operator kind/arity 非法")
        if (not self.payload_kind or not self.query_kind
                or not self.teacher_evidence_kind
                or not self.required_perturbations
                or not self.certificate_kinds):
            raise W07RegistryError("W-07 schema/evidence/certificate 合同为空")
        if (self.consumer_keys != W07_CONSUMER_KEYS
                or self.forming_source_keys != (W07_AUTHORED_SOURCE,)
                or self.teacher_withdrawal_level != 0):
            raise W07RegistryError("W-07 consumer/source/Evidence 等级漂移")

    def arity_for(self, operator_family: str, operator_kind: int) -> int:
        """按 family+kind 返回冻结 arity，错配立即拒绝。"""
        try:
            index = self.operator_families.index(operator_family)
        except ValueError as error:
            raise W07RegistryError("W-07 operator family 未注册") from error
        if self.operator_kinds[index] != operator_kind:
            raise W07RegistryError("W-07 operator family/kind 错配")
        return self.arities[index]


W07_OPERATOR_REGISTRY = {
    "NOT": W07OperatorRegistryEntry(
        "NOT", ("NOT",), (1,), "LogicExecutionQuery",
        "typed_logic_operator_candidate", "NOT_FOUR_STATE_LABEL", (1,),
        (
            "CLOSED_WORLD_CONFUSION", "CONFLICT_SOURCE", "DOUBLE_NEGATION",
            "NONE", "PARSER_REVISION", "PSEUDO_OPERATOR",
        ),
        ("EXPLICIT_STRUCTURAL_NOT_CERTIFICATE",),
    ),
    "AND_OR": W07OperatorRegistryEntry(
        "AND_OR", ("AND", "OR"), (2, 3), "LogicExecutionQuery",
        "typed_logic_operator_candidate", "AND_OR_FOUR_STATE_LABEL", (2, 2),
        (
            "CONFLICT_SOURCE", "NONE", "OPERAND_ORDER_SWAP",
            "OPERATOR_CONFUSION", "PARSER_REVISION",
        ),
        ("ORDERED_CONJUNCTION_CERTIFICATE", "ORDERED_DISJUNCTION_CERTIFICATE"),
    ),
    "CONDITION": W07OperatorRegistryEntry(
        "CONDITION", ("CONDITION",), (4,), "LogicExecutionQuery",
        "typed_logic_operator_candidate", "CONDITION_FOUR_STATE_LABEL", (2,),
        (
            "ANTECEDENT_CONSEQUENT_SWAP", "CAUSAL_CONFUSION",
            "CONFLICT_SOURCE", "NONE", "PARSER_REVISION",
            "TEMPORAL_CONFUSION",
        ),
        ("NECESSARY_SUFFICIENT_CONDITION_CERTIFICATE",),
    ),
    "EXISTS": W07OperatorRegistryEntry(
        "EXISTS", ("EXISTS",), (5,), "QuantifierExecutionQuery",
        "typed_quantifier_candidate", "EXISTS_FINITE_DOMAIN_LABEL", (1,),
        (
            "CONFLICT_SOURCE", "DOMAIN_CLOSURE_CONFUSION",
            "DOMAIN_TYPE_MISMATCH", "EMPTY_DOMAIN_CONFUSION", "NONE",
            "PARSER_REVISION", "QUANTIFIER_SWAP",
        ),
        ("EXISTENTIAL_BINDER_DOMAIN_CERTIFICATE",),
    ),
    "FORALL": W07OperatorRegistryEntry(
        "FORALL", ("FORALL",), (6,), "QuantifierExecutionQuery",
        "typed_quantifier_candidate", "FORALL_FINITE_DOMAIN_LABEL", (1,),
        (
            "CONFLICT_SOURCE", "DOMAIN_CLOSURE_CONFUSION",
            "DOMAIN_TYPE_MISMATCH", "EMPTY_DOMAIN_CONFUSION", "NONE",
            "PARSER_REVISION", "QUANTIFIER_SWAP",
        ),
        ("UNIVERSAL_BINDER_DOMAIN_CERTIFICATE",),
    ),
    "MODAL": W07OperatorRegistryEntry(
        "MODAL", ("MODAL",), (7,), "ModalExecutionQuery",
        "typed_modal_candidate", "MODAL_RESOLVER_LABEL", (1,),
        (
            "BUDGET_UNDECIDED", "CONFLICT_SOURCE", "MODAL_SCOPE_SHIFT",
            "NONE", "PARSER_REVISION", "RESOLVER_DENIED",
            "RESOLVER_MISSING",
        ),
        ("MODAL_WORLD_FRAME_DOMAIN_CERTIFICATE",),
    ),
    "NESTED_SCOPE": W07OperatorRegistryEntry(
        "NESTED_SCOPE", ("NOT", "EXISTS", "FORALL", "MODAL"),
        (1, 5, 6, 7), "NestedScopeExecutionQuery",
        "typed_nested_scope_candidate", "NESTED_SCOPE_LABEL", (1, 1, 1, 1),
        (
            "BUDGET_UNDECIDED", "CONFLICT_SOURCE", "MISSING_INNER_OPERATOR",
            "MODAL_SCOPE_SHIFT", "NONE", "PARSER_REVISION", "QUANTIFIER_SWAP",
        ),
        ("PARENT_CHILD_SCOPE_USE_CERTIFICATE",),
    ),
}


@dataclass(frozen=True)
class W07RegistryAuditReport:
    """汇总七子阶段 train payload 的公开覆盖，不写学习状态。"""

    observation_count: int
    teacher_evidence_count: int
    substage_counts: tuple[tuple[str, int], ...]
    operator_family_counts: tuple[tuple[str, int], ...]
    source_keys: tuple[str, ...]


def _strict_key(value: Any, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W07RegistryError(f"{where} 必须是非空严格整数 key")
    return tuple(value)


def _validate_definition(value: Any, *, arity: int) -> None:
    """验证公开 payload 中 operator definition 的 Role+ordinal 完整性。"""
    if not isinstance(value, dict) or set(value) != {
            "instruction_key", "slots", "structure_key"}:
        raise W07RegistryError("W-07 operator definition 字段漂移")
    _strict_key(value["instruction_key"], where="instruction_key")
    _strict_key(value["structure_key"], where="structure_key")
    slots = value["slots"]
    if not isinstance(slots, list) or len(slots) != arity:
        raise W07RegistryError("W-07 operator Role arity 漂移")
    normalized = []
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {"ordinal", "role_key"}:
            raise W07RegistryError("W-07 operator slot 字段漂移")
        role = _strict_key(slot["role_key"], where="role_key")
        ordinal = slot["ordinal"]
        if type(ordinal) is not int or ordinal < 0:
            raise W07RegistryError("W-07 operator slot ordinal 非法")
        normalized.append((role, ordinal))
    if len(normalized) != len(set(normalized)):
        raise W07RegistryError("W-07 operator Role+ordinal 重复")


def _validate_candidate(value: Any, *, arity: int) -> None:
    """验证 candidate Proposition、competition、source 与图 binding 数。"""
    if not isinstance(value, dict) or set(value) != {
            "candidate_key", "competition_key", "forming_source_keys",
            "graph_binding_count"}:
        raise W07RegistryError("W-07 candidate spec 字段漂移")
    _strict_key(value["candidate_key"], where="candidate_key")
    _strict_key(value["competition_key"], where="competition_key")
    sources = value["forming_source_keys"]
    if not isinstance(sources, list) or not sources:
        raise W07RegistryError("W-07 candidate forming source 为空")
    for item in sources:
        _strict_key(item, where="forming_source_key")
    if value["graph_binding_count"] != arity + 2:
        raise W07RegistryError("W-07 candidate graph binding count 漂移")


def _validate_observation_payload(
        observation: ObservationRecord,
        entry: W07OperatorRegistryEntry,
        ) -> tuple[str, ...]:
    """验证 query kind、operator、Role arity、scope 和 cue 非权威边界。"""
    value = observation.typed_payload.to_value()
    if (not isinstance(value, dict)
            or value.get("query_kind") != entry.query_kind
            or value.get("surface_cue_authoritative") != 0
            or not isinstance(value.get("bound_root"), dict)
            or not isinstance(value.get("candidate_protocol"), dict)
            or not isinstance(value.get("consumer_request"), dict)):
        raise W07RegistryError("W-07 typed payload 基础合同漂移")
    request = value["consumer_request"]
    if (not isinstance(request.get("budget"), dict)
            or not request.get("root_key")
            or not request.get("scope_key")):
        raise W07RegistryError("W-07 consumer request 缺 root/scope/budget")

    if entry.substage_key != "NESTED_SCOPE":
        family = value.get("operator_family")
        kind = value.get("operator_kind")
        if not isinstance(family, str) or type(kind) is not int:
            raise W07RegistryError("W-07 operator family/kind 缺失")
        arity = entry.arity_for(family, kind)
        _validate_definition(value.get("operator_definition"), arity=arity)
        _validate_candidate(value.get("candidate_spec"), arity=arity)
        if entry.substage_key in {"EXISTS", "FORALL"}:
            if not isinstance(value.get("quantifier_definition"), dict):
                raise W07RegistryError("W-07 quantifier 缺 Binder/domain 定义")
        if entry.substage_key == "MODAL":
            if not isinstance(value.get("modal_resolution_plan"), dict):
                raise W07RegistryError("W-07 modal 缺 resolver plan")
        return (family,)

    layers = value.get("layers")
    if not isinstance(layers, list) or len(layers) < 2:
        raise W07RegistryError("W-07 nested scope layers 不完整")
    families = []
    for layer in layers:
        if not isinstance(layer, dict):
            raise W07RegistryError("W-07 nested scope layer 类型错误")
        family = layer.get("operator_family")
        kind = layer.get("operator_kind")
        if not isinstance(family, str) or type(kind) is not int:
            raise W07RegistryError("W-07 nested operator family/kind 缺失")
        arity = entry.arity_for(family, kind)
        _validate_definition(layer.get("operator_definition"), arity=arity)
        _validate_candidate(layer.get("candidate_spec"), arity=arity)
        if layer.get("candidate_available") not in {0, 1}:
            raise W07RegistryError("W-07 nested candidate availability 非 bit")
        families.append(family)
    derivation = value.get("derivation_order")
    if not isinstance(derivation, list) or len(derivation) != len(layers):
        raise W07RegistryError("W-07 nested derivation order 漂移")
    return tuple(families)


def _validate_teacher(
        observation: ObservationRecord,
        teacher: TeacherEvidenceRecord,
        entry: W07OperatorRegistryEntry,
        ) -> str:
    if (teacher.observation_key != observation.stable_key
            or teacher.source_ref_key != observation.source_ref_key
            or teacher.visible_from_stage != "W-07"
            or teacher.withdrawal_level != entry.teacher_withdrawal_level
            or teacher.evidence_kind != entry.teacher_evidence_kind):
        raise W07RegistryError("W-07 TeacherEvidence 引用或种类漂移")
    value = teacher.typed_evidence.to_value()
    if not isinstance(value, dict) or set(value) != {
            "expected_payload", "expected_state", "seed_id"}:
        raise W07RegistryError("W-07 TeacherEvidence 字段集合漂移")
    state = value["expected_state"]
    if state not in W07_EXPECTED_STATES or not isinstance(
            value["expected_payload"], dict):
        raise W07RegistryError("W-07 TeacherEvidence 四态/payload 非法")
    return state


def _validate_supersede_chain(
        observation: ObservationRecord,
        observations: dict[StableRecordKey, ObservationRecord],
        ) -> None:
    target = observation.supersedes_key
    is_revision = observation.perturbation_kind == "PARSER_REVISION"
    if (target is None) != (not is_revision):
        raise W07RegistryError("W-07 parser revision 与 supersede 标记不一致")
    if target is None:
        return
    prior = observations.get(target)
    if (prior is None or prior.substage != observation.substage
            or prior.logical_order >= observation.logical_order
            or prior.supersedes_key is not None):
        raise W07RegistryError("W-07 supersede target 缺失、跨阶段或顺序非法")


def audit_w07_registry_payload(payload: W07TrainingPayload) -> W07RegistryAuditReport:
    """审核七个 forming pack 的 train schema、四态 Evidence 与 lifecycle。"""
    if not isinstance(payload, W07TrainingPayload):
        raise TypeError("W-07 registry audit 需要 W07TrainingPayload")
    sources = {item.stable_key: item for item in payload.source_refs}
    observations = {item.stable_key: item for item in payload.observations}
    teachers: dict[StableRecordKey, list[TeacherEvidenceRecord]] = defaultdict(list)
    for teacher in payload.teacher_evidence:
        teachers[teacher.observation_key].append(teacher)
    if not observations or set(teachers) != set(observations):
        raise W07RegistryError("W-07 Observation/Evidence inventory 不闭合")

    substage_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    roles: dict[str, set[str]] = defaultdict(set)
    perturbations: dict[str, set[str]] = defaultdict(set)
    states: dict[str, set[str]] = defaultdict(set)
    source_keys = set()
    for key, observation in observations.items():
        entry = W07_OPERATOR_REGISTRY.get(observation.substage)
        source = sources.get(observation.source_ref_key)
        evidence = teachers[key]
        if (entry is None or source is None or len(evidence) != 1):
            raise W07RegistryError("W-07 registry 缺 substage/source/唯一 Evidence")
        if (observation.w_stage != "W-07"
                or observation.split != "train"
                or observation.epistemic_role != "forming"
                or observation.payload_kind != entry.payload_kind
                or observation.sample_role not in W07_REQUIRED_SAMPLE_ROLES
                or observation.perturbation_kind not in entry.required_perturbations
                or source.source_key not in entry.forming_source_keys
                or source.license_id != W07_LICENSE_ID
                or observation.license_partition != W07_LICENSE_ID):
            raise W07RegistryError("W-07 train Observation 未匹配 registry")
        _validate_supersede_chain(observation, observations)
        families = _validate_observation_payload(observation, entry)
        state = _validate_teacher(observation, evidence[0], entry)
        substage_counts[entry.substage_key] += 1
        family_counts.update(families)
        roles[entry.substage_key].add(observation.sample_role)
        perturbations[entry.substage_key].add(observation.perturbation_kind)
        states[entry.substage_key].add(state)
        source_keys.add(source.source_key)

    if tuple(item for item in W07_SUBSTAGE_ORDER if item in substage_counts) != (
            W07_SUBSTAGE_ORDER):
        raise W07RegistryError("W-07 registry 子阶段覆盖漂移")
    for substage, entry in W07_OPERATOR_REGISTRY.items():
        if (roles[substage] != set(W07_REQUIRED_SAMPLE_ROLES)
                or perturbations[substage] != set(entry.required_perturbations)
                or states[substage] != set(W07_EXPECTED_STATES)
                or not set(entry.operator_families).issubset(family_counts)):
            raise W07RegistryError("W-07 lifecycle/四态/负例/operator 覆盖不完整")
    return W07RegistryAuditReport(
        len(observations),
        sum(len(items) for items in teachers.values()),
        tuple((item, substage_counts[item]) for item in W07_SUBSTAGE_ORDER),
        tuple(sorted(family_counts.items())),
        tuple(sorted(source_keys)),
    )


__all__ = [
    "W07_CONSUMER_KEYS",
    "W07_EXPECTED_STATES",
    "W07_OPERATOR_REGISTRY",
    "W07_REQUIRED_SAMPLE_ROLES",
    "W07OperatorRegistryEntry",
    "W07RegistryAuditReport",
    "W07RegistryError",
    "audit_w07_registry_payload",
]
